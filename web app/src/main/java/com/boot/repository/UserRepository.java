package com.boot.repository;
import javax.transaction.Transactional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.boot.entity.User;

@Transactional
@Repository
public interface UserRepository extends JpaRepository<User, Integer>,JpaSpecificationExecutor<User>{

	@Query(value=" select t from User t where t.username =?1")
	User findByUserName(String username)throws Exception;
	
	@Query(value="select count(1) from t_user where username=?1",nativeQuery=true)
	long countByUserName(String username);
	
	@Query(value="select count(1) from t_user where type=?1 and state<>'2'",nativeQuery=true)
	int countUserRole(Integer roleId);
	
	@Query(value="select count(1) from t_user where department=?1 and state<>'2'",nativeQuery=true)
	int countUserDepartment(Integer departmentId);
}