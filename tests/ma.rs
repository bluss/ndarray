use ndarray::{array};
use ndarray::ma;

#[cfg(test)]
mod test_array_mask {
    use super::*;

    #[test]
    fn test_iter() {
        let data = array![1, 2, 3, 4];
        let mask = array![true, false, true, false];
        let arr = ma::array(data, mask);
        let actual_vec: Vec<_> = arr.iter().collect();
        let expected_vec = vec![
            ma::Masked::Value(&1),
            ma::Masked::Empty,
            ma::Masked::Value(&3),
            ma::Masked::Empty,
        ];
        assert_eq!(actual_vec, expected_vec);
    }

    #[test]
    fn test_compressed() {
        let arr = ma::array(array![1, 2, 3, 4], array![true, true, false, false]);
        let res = arr.compressed();
        assert_eq!(res, array![1, 2]);
    }

    #[test]
    fn test_add() {
        let arr1 = ma::array(array![1, 2, 3, 4], array![true, false, true, false]);
        let arr2 = ma::array(array![4, 3, 2, 1], array![true, false, false, false]);
        let res = arr1 + arr2;
        let actual_vec: Vec<_> = res.iter().collect();
        let expected_vec = vec![
            ma::Masked::Value(&5),
            ma::Masked::Empty,
            ma::Masked::Empty,
            ma::Masked::Empty,
        ];
        assert_eq!(actual_vec, expected_vec);
    }
}
